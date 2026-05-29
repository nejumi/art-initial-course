from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Any

from course.shared.config import config_from_env
from course.shared.schemas import RetailScenario
from course.shared.rollout import rollout_retail


def make_weave_model(model: Any, *, artifact_uri: str | None = None):
    import weave

    class RetailARTModel(weave.Model):
        artifact_uri: str | None = None

        @weave.op()
        async def predict(self, scenario: dict[str, Any]) -> dict[str, Any]:
            traj = await rollout_retail(model, RetailScenario(**scenario), split="eval", temperature=0.2)
            metadata = dict(traj.metadata)
            metadata["artifact_uri"] = self.artifact_uri
            return {"reward": traj.reward, "metrics": traj.metrics, "metadata": metadata}

    return RetailARTModel(artifact_uri=artifact_uri)


if __name__ == "__main__":
    cfg = config_from_env()
    print("Import make_weave_model(model, artifact_uri=...) from this module.")
    print("Project:", cfg.project)
