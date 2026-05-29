from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from course.shared.data import load_cached_split, scenarios_from_records, write_sample_dataset
from course.shared.rewards import score_messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect retail scenarios and reward scoring without training.")
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / f"{args.split}.jsonl").exists():
        write_sample_dataset(data_dir)

    records = load_cached_split(data_dir, args.split, limit=args.limit)
    scenarios = scenarios_from_records(records, split=args.split)
    for scenario in scenarios:
        result = score_messages(scenario.reference_messages, scenario)
        print("=" * 80)
        print("scenario:", scenario.id)
        print("user:", scenario.user_message)
        print("expected_tools:", scenario.expected_tool_names)
        print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
