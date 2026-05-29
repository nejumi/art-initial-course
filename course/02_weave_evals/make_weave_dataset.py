from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from course.shared.data import load_cached_split, scenarios_from_records, write_jsonl, write_sample_dataset
from course.shared.tracing import init_weave


def main() -> None:
    parser = argparse.ArgumentParser(description="Create local and optional Weave retail eval dataset.")
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--name", default="tau-retail-holdout")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / f"{args.split}.jsonl").exists():
        write_sample_dataset(data_dir)

    records = load_cached_split(data_dir, args.split, limit=args.limit)
    rows = []
    for scenario in scenarios_from_records(records, split=args.split):
        rows.append(
            {
                "scenario": scenario.to_dict(),
                "target": {
                    "expected_tool_names": scenario.expected_tool_names,
                    "expected_final_text": scenario.expected_final_text,
                },
            }
        )
    output = data_dir / "weave_eval_rows.jsonl"
    write_jsonl(output, rows)
    print(f"Wrote {len(rows)} eval rows to {output}")

    if args.publish:
        init_weave()
        import weave
        dataset = weave.Dataset(name=args.name, rows=rows)
        ref = weave.publish(dataset)
        print("Published Weave dataset:", ref)


if __name__ == "__main__":
    main()
