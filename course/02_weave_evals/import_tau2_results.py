from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from course.shared.data import write_jsonl


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def bool_metric(value: Any) -> float:
    return 1.0 if value is True else 0.0


def reward_breakdown_value(reward_info: dict[str, Any], *names: str) -> float | None:
    breakdown = reward_info.get("reward_breakdown") or {}
    for name in names:
        if name in breakdown:
            try:
                return float(breakdown[name])
            except Exception:
                return None
    return None


def checks_success(reward_info: dict[str, Any], *names: str, default: float = 1.0) -> float:
    for name in names:
        checks = reward_info.get(name)
        if isinstance(checks, list):
            if not checks:
                return default
            return bool_metric(all((check or {}).get("met") for check in checks if isinstance(check, dict)))
    return default


def partial_action_metrics(reward_info: dict[str, Any]) -> dict[str, float]:
    checks = reward_info.get("action_checks") or []
    if not isinstance(checks, list) or not checks:
        return {}
    total = 0
    met = 0
    read_total = 0
    read_met = 0
    write_total = 0
    write_met = 0
    for check in checks:
        if not isinstance(check, dict):
            continue
        total += 1
        is_met = bool(check.get("met"))
        met += int(is_met)
        tool_type = str(check.get("tool_type") or "").lower()
        if tool_type == "read":
            read_total += 1
            read_met += int(is_met)
        elif tool_type == "write":
            write_total += 1
            write_met += int(is_met)
    metrics = {"tau2_action_match": met / total if total else 0.0}
    if read_total:
        metrics["tau2_read_action_match"] = read_met / read_total
    if write_total:
        metrics["tau2_write_action_match"] = write_met / write_total
    return metrics


def row_from_simulation(simulation: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    reward_info = simulation.get("reward_info") or {}
    reward = float(reward_info.get("reward") or 0.0)
    db_reward = reward_breakdown_value(reward_info, "DB", "db")
    communicate_reward = reward_breakdown_value(reward_info, "COMMUNICATE", "communicate")
    nl_assertion_reward = reward_breakdown_value(reward_info, "NL_ASSERTION", "nl_assertion", "NL_ASSERTIONS", "nl_assertions")
    metrics: dict[str, float] = {
        "tau2_official_success": reward,
        "tau2_official_reward": reward,
        "tau2_db_success": db_reward if db_reward is not None else bool_metric(nested(reward_info, "db_check", "db_match")),
        "tau2_communicate_success": communicate_reward
        if communicate_reward is not None
        else checks_success(reward_info, "communicate_checks", "communication_checks"),
        "tau2_nl_assertion_success": nl_assertion_reward
        if nl_assertion_reward is not None
        else checks_success(reward_info, "nl_assertion_checks", "nl_assertions", default=1.0),
        "tau2_terminated_cleanly": 1.0
        if simulation.get("termination_reason") in {"agent_stop", "user_stop", "AGENT_STOP", "USER_STOP"}
        else 0.0,
    }
    metrics.update(partial_action_metrics(reward_info))
    return {
        "scenario_id": str(simulation.get("task_id") or simulation.get("id") or ""),
        "reward": reward,
        "metrics": metrics,
        "logs": {
            "source_path": str(source_path),
            "termination_reason": simulation.get("termination_reason"),
            "reward_basis": reward_info.get("reward_basis"),
            "reward_breakdown": reward_info.get("reward_breakdown"),
        },
    }


def convert_results(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    simulations = data.get("simulations") or []
    if not isinstance(simulations, list):
        raise ValueError(f"Expected a tau2 Results JSON with a simulations list: {path}")
    return [row_from_simulation(simulation, source_path=path) for simulation in simulations if isinstance(simulation, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official tau2 Results JSON into the course eval JSONL format."
    )
    parser.add_argument("paths", nargs="+", help="tau2 Results JSON files after official reward recomputation.")
    parser.add_argument("--output", required=True, help="Output JSONL path for compare_checkpoints.py.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path_str in args.paths:
        rows.extend(convert_results(Path(path_str)))
    count = write_jsonl(args.output, rows)
    print({"output": args.output, "rows": count, "inputs": args.paths})


if __name__ == "__main__":
    main()
