from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Message = dict[str, Any]
Tool = dict[str, Any]


@dataclass
class RetailScenario:
    id: str
    split: Literal["train", "validation", "test", "val"]
    system_message: str
    user_message: str
    tools: list[Tool]
    reference_messages: list[Message]
    expected_tool_names: list[str]
    expected_final_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    max_turns: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvStep:
    tool_messages: list[Message] = field(default_factory=list)
    user_messages: list[Message] = field(default_factory=list)
    done: bool = False
    invalid_tool_calls: int = 0
    expected_tool_names: list[str] = field(default_factory=list)
    actual_tool_names: list[str] = field(default_factory=list)


@dataclass
class RewardResult:
    reward: float
    metrics: dict[str, float]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {"reward": self.reward, "metrics": self.metrics, "explanation": self.explanation}
