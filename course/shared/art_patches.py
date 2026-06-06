from __future__ import annotations

import json
from typing import Any


def install_art_tool_argument_normalizer() -> bool:
    """Patch ART's chat-template argument normalizer for mapping-style templates.

    Some chat templates, including LiquidAI LFM2 variants, iterate over
    ``tool_call.function.arguments.items()``. ART already normalizes JSON-string
    arguments for one template spelling, but these variants need the same
    treatment. Keep this course-local and idempotent so it is easy to remove once
    upstream ART covers the broader pattern.
    """

    try:
        from art.preprocessing import tokenize
    except Exception:
        return False

    current = getattr(tokenize, "_normalize_tool_call_arguments_for_chat_template", None)
    if current is None or getattr(current, "_retail_course_patched", False):
        return current is not None

    def _template_wants_argument_mapping(chat_template: str) -> bool:
        return any(
            marker in chat_template
            for marker in (
                "tool_call.arguments|items",
                "tool_call.arguments.items",
                "tool_call.function.arguments.items",
                "func_args.items",
                ".arguments.items()",
                "arguments.items()",
            )
        )

    def _normalize_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"_raw": raw}
            return parsed if isinstance(parsed, dict) else {"_value": parsed}
        if raw is None:
            return {}
        return {"_raw": raw}

    def _patched(tokenizer: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chat_template = getattr(tokenizer, "chat_template", None)
        if not isinstance(chat_template, str) or not _template_wants_argument_mapping(chat_template):
            return messages

        normalized_messages: list[dict[str, Any]] = []
        for message in messages:
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                normalized_messages.append(message)
                continue

            normalized_tool_calls: list[Any] = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    normalized_tool_calls.append(tool_call)
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    normalized_tool_calls.append(tool_call)
                    continue
                normalized_tool_calls.append(
                    {
                        **tool_call,
                        "function": {
                            **function,
                            "arguments": _normalize_arguments(function.get("arguments")),
                        },
                    }
                )

            normalized_messages.append({**message, "tool_calls": normalized_tool_calls})
        return normalized_messages

    _patched._retail_course_patched = True  # type: ignore[attr-defined]
    tokenize._normalize_tool_call_arguments_for_chat_template = _patched
    return True
