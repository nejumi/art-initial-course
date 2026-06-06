from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .schemas import EnvStep, Message, RetailScenario


STATE_CHANGING_TOOL_NAMES = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_order",
    "modify_user_address",
    "return_delivered_order",
    "return_delivered_order_items",
    "transfer_to_human_agents",
}


def is_state_changing_tool(name: str) -> bool:
    return name in STATE_CHANGING_TOOL_NAMES


def tool_names_from_schema(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") or {}
        name = function.get("name")
        if name:
            names.add(str(name))
    return names


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


def _canonical_argument_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return value
    if isinstance(value, list):
        return [_canonical_argument_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical_argument_value(inner) for key, inner in sorted(value.items())}
    return str(value).strip().lower()


def argument_value_matches(actual: Any, expected: Any) -> bool:
    actual_value = _canonical_argument_value(actual)
    expected_value = _canonical_argument_value(expected)
    if isinstance(actual_value, int | float) and isinstance(expected_value, int | float):
        return abs(float(actual_value) - float(expected_value)) <= 1e-6
    return actual_value == expected_value


def tool_argument_match_score(actual_call: dict[str, Any], expected_call: dict[str, Any]) -> float:
    actual = {str(key): value for key, value in tool_call_arguments(actual_call).items()}
    expected = {str(key): value for key, value in tool_call_arguments(expected_call).items()}
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0

    hits = 0
    for key, expected_value in expected.items():
        if key in actual and argument_value_matches(actual[key], expected_value):
            hits += 1
    precision = hits / len(actual)
    recall = hits / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def tool_call_matches(actual_call: dict[str, Any], expected_call: dict[str, Any]) -> bool:
    return tool_call_name(actual_call) == tool_call_name(expected_call) and tool_argument_match_score(actual_call, expected_call) == 1.0


def find_matching_reference_call(
    actual_call: dict[str, Any],
    expected_calls: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]] | None:
    for index, expected_call in enumerate(expected_calls):
        if tool_call_matches(actual_call, expected_call):
            return index, expected_call
    return None


def extract_tool_calls_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in messages:
        msg = choice_to_message(item)
        if msg.get("role") != "assistant":
            continue
        calls.extend(dict(call) for call in msg.get("tool_calls") or [])
    return calls


def extract_tool_names_from_messages(messages: list[Any]) -> list[str]:
    names: list[str] = []
    for call in extract_tool_calls_from_messages(messages):
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

        assistant_chunks: list[Message] = []
        while i < len(reference_messages) and reference_messages[i].get("role") == "assistant":
            assistant_chunks.append(reference_messages[i])
            i += 1
        assistant_message = merge_assistant_chunks(assistant_chunks)
        expected_tool_names = [tool_call_name(call) for call in assistant_message.get("tool_calls") or []]

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


def merge_assistant_chunks(chunks: list[Message]) -> Message:
    if not chunks:
        return {"role": "assistant", "content": ""}
    merged = dict(chunks[-1])
    content_parts = [str(chunk.get("content") or "").strip() for chunk in chunks if chunk.get("content")]
    tool_calls: list[dict[str, Any]] = []
    for chunk in chunks:
        tool_calls.extend(dict(call) for call in chunk.get("tool_calls") or [])
    merged["role"] = "assistant"
    merged["content"] = "\n".join(part for part in content_parts if part) or None
    if tool_calls:
        merged["tool_calls"] = tool_calls
    else:
        merged.pop("tool_calls", None)
    return merged


class ReplayRetailEnv:
    """A small replay-style environment built from open SFT trajectories.

    This is intentionally simpler than the full tau2 retail environment. It lets the
    course teach ART trajectory construction, tool messages, and verifiable rewards
    before students swap in a full benchmark environment.
    """

    def __init__(
        self,
        scenario: RetailScenario,
        *,
        terminate_on_invalid: bool = True,
        strict_reference_actions: bool = True,
        allow_reference_state_action_jumps: bool = False,
    ) -> None:
        self.scenario = scenario
        self.terminate_on_invalid = terminate_on_invalid
        self.strict_reference_actions = strict_reference_actions
        self.allow_reference_state_action_jumps = allow_reference_state_action_jumps
        self.turns = parse_reference_turns(scenario.reference_messages)
        self.turn_index = 0
        self.invalid_tool_calls = 0
        self.invalid_state_mutations = 0
        self.read_only_reference_mismatches = 0
        self.unknown_tool_calls = 0
        self.bad_read_only_calls = 0
        self.bad_state_actions = 0
        self.missing_state_actions = 0
        self.terminated_on_invalid = False
        self.actual_tool_names: list[str] = []
        self.valid_tool_names = tool_names_from_schema(scenario.tools)
        self.accepted_reference_state_output_indices: set[int] = set()
        self.reference_tool_outputs: list[tuple[int, dict[str, Any], str]] = []
        for turn_index, turn in enumerate(self.turns):
            calls = list(turn.assistant_message.get("tool_calls") or [])
            for index, call in enumerate(calls):
                content = "{}"
                if index < len(turn.tool_messages):
                    content = str(turn.tool_messages[index].get("content") or "{}")
                self.reference_tool_outputs.append((turn_index, dict(call), content))

    def _can_accept_reference_state_jump(self, reference_output_index: int) -> bool:
        if not self.allow_reference_state_action_jumps or self.strict_reference_actions:
            return False
        if reference_output_index in self.accepted_reference_state_output_indices:
            return False
        if reference_output_index >= len(self.reference_tool_outputs):
            return False
        matched_turn_index, matched_call, _content = self.reference_tool_outputs[reference_output_index]
        if matched_turn_index < self.turn_index or not is_state_changing_tool(tool_call_name(matched_call)):
            return False
        for prior_index, (prior_turn_index, prior_call, _prior_content) in enumerate(self.reference_tool_outputs):
            if prior_index >= reference_output_index:
                break
            if prior_turn_index < self.turn_index:
                continue
            if is_state_changing_tool(tool_call_name(prior_call)) and prior_index not in self.accepted_reference_state_output_indices:
                return False
        return True

    def step(self, assistant_message: Any) -> EnvStep:
        msg = message_to_dict(assistant_message)
        calls = tool_calls_from_message(msg)
        expected = self.turns[self.turn_index] if self.turn_index < len(self.turns) else None
        expected_tool_calls = list(expected.assistant_message.get("tool_calls") or []) if expected else []
        expected_tool_names = [tool_call_name(call) for call in expected_tool_calls]
        tool_messages: list[Message] = []
        invalid = 0
        invalid_state_mutations = 0
        read_only_reference_mismatches = 0
        unknown_tool_calls = 0
        bad_read_only_calls = 0
        bad_state_actions = 0
        matched_expected_tools = bool(expected) and len(calls) == len(expected_tool_names)
        accepted_state_jump_turn_index: int | None = None
        accepted_state_jump_output_index: int | None = None
        skipped_reference_turns_before_state_action = 0

        for index, call in enumerate(calls):
            name = tool_call_name(call)
            self.actual_tool_names.append(name)
            expected_call = expected_tool_calls[index] if index < len(expected_tool_calls) else None
            expected_name = expected_tool_names[index] if index < len(expected_tool_names) else None
            arguments_match = bool(expected_call) and tool_argument_match_score(call, expected_call) == 1.0
            known_tool = bool(name) and (not self.valid_tool_names or name in self.valid_tool_names)
            state_changing = is_state_changing_tool(name)
            reference_match = find_matching_reference_call(call, [item[1] for item in self.reference_tool_outputs])
            if not known_tool:
                invalid += 1
                unknown_tool_calls += 1
                matched_expected_tools = False
                content = json.dumps(
                    {
                        "error": "unknown_tool_call",
                        "called": name,
                        "expected": expected_name,
                        "available_tools": sorted(self.valid_tool_names),
                        "arguments": tool_call_arguments(call),
                    }
                )
            elif name == expected_name and arguments_match and expected and index < len(expected.tool_messages):
                content = expected.tool_messages[index].get("content") or "{}"
            elif reference_match is not None and not self.strict_reference_actions and not state_changing:
                # Outcome-style tau-bench scoring does not require reproducing the
                # unique reference path. If the model calls any known reference
                # read-only tool with exact arguments, return the recorded
                # observation but keep scripted user progression unchanged.
                # State-changing calls are handled separately: only a single
                # exact reference mutation may advance the replay cursor.
                content = self.reference_tool_outputs[reference_match[0]][2]
                matched_expected_tools = False
            elif (
                reference_match is not None
                and state_changing
                and len(calls) == 1
                and self._can_accept_reference_state_jump(reference_match[0])
            ):
                # Tau-style scoring cares about the target state mutation, not
                # exact read-only replay. For the lightweight replay environment,
                # allow a single exact reference state-changing action to skip
                # preceding read-only turns while still rejecting wrong mutations.
                matched_turn_index, _matched_call, content = self.reference_tool_outputs[reference_match[0]]
                self.accepted_reference_state_output_indices.add(reference_match[0])
                accepted_state_jump_turn_index = matched_turn_index
                accepted_state_jump_output_index = reference_match[0]
                skipped_reference_turns_before_state_action = max(0, matched_turn_index - self.turn_index)
                matched_expected_tools = False
            else:
                if self.strict_reference_actions or state_changing:
                    invalid += 1
                if state_changing:
                    invalid_state_mutations += 1
                    bad_state_actions += 1
                    error = "unexpected_state_action"
                else:
                    read_only_reference_mismatches += 1
                    bad_read_only_calls += 1
                    error = "unexpected_read_only_tool_call"
                matched_expected_tools = False
                content = json.dumps(
                    {
                        "error": error,
                        "called": name,
                        "expected": expected_name,
                        "argument_match_score": tool_argument_match_score(call, expected_call) if expected_call else 0.0,
                        "arguments": tool_call_arguments(call),
                    }
                )
            tool_messages.append({"role": "tool", "tool_call_id": tool_call_id(call, index), "content": content})

        missing_tool_calls = max(0, len(expected_tool_names) - len(calls))
        missing_state_actions = 0
        if not calls:
            missing_state_actions = sum(
                1 for expected_call in expected_tool_calls if is_state_changing_tool(tool_call_name(expected_call))
            )
        if self.strict_reference_actions:
            invalid += missing_tool_calls
        self.invalid_tool_calls += invalid
        self.invalid_state_mutations += invalid_state_mutations
        self.read_only_reference_mismatches += read_only_reference_mismatches
        self.unknown_tool_calls += unknown_tool_calls
        self.bad_read_only_calls += bad_read_only_calls
        self.bad_state_actions += bad_state_actions
        self.missing_state_actions += missing_state_actions
        user_messages: list[Message] = []

        terminated_on_invalid = self.terminate_on_invalid and invalid > 0
        if terminated_on_invalid:
            self.terminated_on_invalid = True

        if expected is None:
            done = True
        elif terminated_on_invalid:
            done = True
        elif accepted_state_jump_turn_index is not None and invalid == 0:
            user_messages = list(self.turns[accepted_state_jump_turn_index].user_messages)
            self.turn_index = max(self.turn_index, accepted_state_jump_turn_index + 1)
            done = self.turn_index >= len(self.turns)
        elif expected_tool_names:
            if matched_expected_tools and invalid == 0:
                user_messages = list(expected.user_messages)
                self.turn_index += 1
                done = self.turn_index >= len(self.turns) or (not calls and not user_messages)
            elif calls:
                done = False
            else:
                done = True
        elif calls:
            done = False
        else:
            user_messages = list(expected.user_messages)
            self.turn_index += 1
            done = self.turn_index >= len(self.turns) or not user_messages

        return EnvStep(
            tool_messages=tool_messages,
            user_messages=user_messages,
            done=done,
            invalid_tool_calls=invalid,
            invalid_state_mutations=invalid_state_mutations,
            read_only_reference_mismatches=read_only_reference_mismatches,
            unknown_tool_calls=unknown_tool_calls,
            bad_read_only_calls=bad_read_only_calls,
            bad_state_actions=bad_state_actions,
            missing_state_actions=missing_state_actions,
            terminated_on_invalid=terminated_on_invalid,
            expected_tool_names=expected_tool_names,
            actual_tool_names=[tool_call_name(call) for call in calls],
            accepted_state_action_jump=accepted_state_jump_turn_index is not None and invalid == 0,
            accepted_reference_state_output_index=accepted_state_jump_output_index,
            accepted_reference_state_turn_index=accepted_state_jump_turn_index,
            skipped_reference_turns_before_state_action=skipped_reference_turns_before_state_action,
        )
