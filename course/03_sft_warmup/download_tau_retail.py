from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from course.shared.config import DEFAULT_DATASET_ID, ensure_data_dir
from course.shared.data import SPLITS, load_hf_split, write_jsonl, write_sample_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache the open retail SFT dataset.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--output-dir", default="data/retail")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--sample-if-download-fails", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_data_dir(Path(args.output_dir))
    if args.sample_only:
        write_sample_dataset(output_dir)
        print(f"Wrote smoke sample data to {output_dir}")
        return

    metadata = {"dataset_id": args.dataset_id, "splits": {}, "source": "huggingface"}
    try:
        for split in SPLITS:
            rows = load_hf_split(args.dataset_id, split, limit=args.limit)
            count = write_jsonl(output_dir / f"{split}.jsonl", rows)
            metadata["splits"][split] = count
            print(f"{split}: {count} rows")
    except Exception as exc:
        if not args.sample_if_download_fails:
            raise
        print("Download failed; writing smoke sample instead:", exc)
        write_sample_dataset(output_dir)
        metadata = {"dataset_id": args.dataset_id, "source": "course-smoke-sample", "error": str(exc)}

    (output_dir / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata to {output_dir / 'source_metadata.json'}")


if __name__ == "__main__":
    main()
