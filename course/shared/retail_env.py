from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .schemas import EnvStep, Message, RetailScenario


def message_to_dict(message: Any) -> Message:
    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    if hasattr(message, "to_dict"):
        return message.to_dict()
    raise TypeError(f"Unsupported message type: {type(message)!r}")


def choice_to_message(choice: Any) -> Message:
    if isinstance(choice, dict) and "message" in choice:
        return message_to_dict(choice["message"])
    if hasattr(choice, "message"):
        return message_to_dict(choice.message)
    return message_to_dict(choice)


def tool_calls_from_message(message: Any) -> list[dict[str, Any]]:
    msg = message_to_dict(message)
    calls = msg.get("tool_calls") or []
    return [dict(call) for call in calls]


def tool_call_name(call: dict[str, Any]) -> str:
    function = call.get("function") or {}
    return str(function.get("name") or "")


def tool_call_id(call: dict[str, Any], index: int) -> str:
    return str(call.get("id") or f"call_{index}")


def tool_call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = (call.get("function") or {}).get("arguments")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {"_raw": raw}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    except json.JSONDecodeError:
        return {"_raw": raw}


def extract_tool_names_from_messages(messages: list[Any]) -> list[str]:
    names: list[str] = []
    for item in messages:
        msg = choice_to_message(item) if hasattr(item, "message") else message_to_dict(item)
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            name = tool_call_name(call)
            if name:
                names.append(name)
    return names


def first_message_content(messages: list[Message], role: str) -> str:
    for msg in messages:
        if msg.get("role") == role:
            return str(msg.get("content") or "")
    return ""


@dataclass
class ReferenceAssistantTurn:
    assistant_message: Message
    expected_tool_names: list[str]
    tool_messages: list[Message] = field(default_factory=list)
    user_messages: list[Message] = field(default_factory=list)


def parse_reference_turns(reference_messages: list[Message]) -> list[ReferenceAssistantTurn]:
    turns: list[ReferenceAssistantTurn] = []
    i = 0
    while i < len(reference_messages):
        msg = reference_messages[i]
        if msg.get("role") != "assistant":
            i += 1
            continue
        assistant_message = msg
        expected_tool_names = [tool_call_name(call) for call in msg.get("tool_calls") or []]
        i += 1
        tool_messages: list[Message] = []
        user_messages: list[Message] = []
        while i < len(reference_messages) and reference_messages[i].get("role") != "assistant":
            role = reference_messages[i].get("role")
            if role == "tool":
                tool_messages.append(reference_messages[i])
            elif role == "user":
                user_messages.append(reference_messages[i])
            i += 1
        turns.append(
            ReferenceAssistantTurn(
                assistant_message=assistant_message,
                expected_tool_names=[name for name in expected_tool_names if name],
                tool_messages=tool_messages,
                user_messages=user_messages,
            )
        )
    return turns


class ReplayRetailEnv:
    """A small replay-style environment built from open SFT trajectories.

    This is intentionally simpler than the full tau2 retail environment. It lets the
    course teach ART trajectory construction, tool messages, and verifiable rewards
    before students swap in a full benchmark environment.
    """

    def __init__(self, scenario: RetailScenario) -> None:
        self.scenario = scenario
        self.turns = parse_reference_turns(scenario.reference_messages)
        self.turn_index = 0
        self.invalid_tool_calls = 0
        self.actual_tool_names: list[str] = []

    def step(self, assistant_message: Any) -> EnvStep:
        msg = message_to_dict(assistant_message)
        calls = tool_calls_from_message(msg)
        expected = self.turns[self.turn_index] if self.turn_index < len(self.turns) else None
        expected_tool_names = expected.expected_tool_names if expected else []
        tool_messages: list[Message] = []
        invalid = 0

        for index, call in enumerate(calls):
            name = tool_call_name(call)
            self.actual_tool_names.append(name)
            expected_name = expected_tool_names[index] if index < len(expected_tool_names) else None
            if name and name == expected_name and expected and index < len(expected.tool_messages):
                content = expected.tool_messages[index].get("content") or "{}"
            else:
                invalid += 1
                content = json.dumps(
                    {
                        "error": "unexpected_tool_call",
                        "called": name,
                        "expected": expected_name,
                        "arguments": tool_call_arguments(call),
                    }
                )
            tool_messages.append({"role": "tool", "tool_call_id": tool_call_id(call, index), "content": content})

        self.invalid_tool_calls += invalid
        user_messages = list(expected.user_messages) if expected else []
        self.turn_index += 1
        done = self.turn_index >= len(self.turns) or (not calls and not user_messages)
        return EnvStep(
            tool_messages=tool_messages,
            user_messages=user_messages,
            done=done,
            invalid_tool_calls=invalid,
            expected_tool_names=expected_tool_names,
            actual_tool_names=[tool_call_name(call) for call in calls],
        )
