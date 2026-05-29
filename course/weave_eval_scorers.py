from course.shared.tracing import weave_op
from typing import Any


def _metric(model_output: dict[str, Any], name: str, default: float = 0.0) -> float:
    metrics = model_output.get("metrics") or {}
    try:
        return float(metrics.get(name, default))
    except Exception:
        return default


@weave_op("task_success_scorer")
def task_success_scorer(model_output: dict[str, Any], **_: Any) -> dict[str, float]:
    return {"task_success": _metric(model_output, "task_success")}


@weave_op("tool_quality_scorer")
def tool_quality_scorer(model_output: dict[str, Any], **_: Any) -> dict[str, float]:
    return {
        "tool_name_f1": _metric(model_output, "tool_name_f1"),
        "tool_order_match": _metric(model_output, "tool_order_match"),
        "invalid_tool_call": _metric(model_output, "invalid_tool_call"),
    }


@weave_op("response_quality_scorer")
def response_quality_scorer(model_output: dict[str, Any], **_: Any) -> dict[str, float]:
    return {
        "final_text_f1": _metric(model_output, "final_text_f1"),
        "has_final_response": _metric(model_output, "has_final_response"),
        "turn_count": _metric(model_output, "turn_count"),
    }


SCORERS = [task_success_scorer, tool_quality_scorer, response_quality_scorer]
