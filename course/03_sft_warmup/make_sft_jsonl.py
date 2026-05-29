from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from course.shared.data import SPLITS, load_cached_split, write_sample_dataset, write_sft_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cached retail records to ART SFT JSONL.")
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write-sample-if-empty", action="store_true", default=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.write_sample_if_empty and not (data_dir / "train.jsonl").exists():
        write_sample_dataset(data_dir)

    for split in SPLITS:
        rows = load_cached_split(data_dir, split, limit=args.limit)
        if not rows:
            continue
        output = data_dir / f"sft_{split}.jsonl"
        count = write_sft_jsonl(rows, output)
        print(f"{output}: {count} examples")


if __name__ == "__main__":
    main()
