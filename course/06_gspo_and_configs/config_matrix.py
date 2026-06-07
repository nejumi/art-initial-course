from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

CONFIGS = [
    {"name": "default-token", "importance_sampling_level": "token", "learning_rate": 5e-6},
    {"name": "gspo-sequence", "importance_sampling_level": "sequence", "learning_rate": 5e-6},
    {"name": "geometric-average", "importance_sampling_level": "geometric_average", "learning_rate": 5e-6},
    {"name": "with-kl", "importance_sampling_level": "token", "learning_rate": 5e-6, "kl_penalty_coef": 0.01},
    {
        "name": "rollout-logprob-diagnostic",
        "importance_sampling_level": "token",
        "learning_rate": 5e-6,
        "notes": "Enable precalculate_logprobs only after confirming backend support in the pinned ART version.",
    },
]


def main() -> None:
    print(json.dumps(CONFIGS, indent=2))


if __name__ == "__main__":
    main()
