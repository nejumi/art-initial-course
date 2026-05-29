from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from course.shared.data import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local eval JSONL files from multiple checkpoints.")
    parser.add_argument("paths", nargs="+", help="JSONL files with reward/metrics rows")
    args = parser.parse_args()
    for path_str in args.paths:
        path = Path(path_str)
        rows = read_jsonl(path)
        if not rows:
            print(path, "no rows")
            continue
        metrics: dict[str, list[float]] = {}
        for row in rows:
            for key, value in (row.get("metrics") or {}).items():
                try:
                    metrics.setdefault(key, []).append(float(value))
                except Exception:
                    pass
        summary = {key: sum(values) / len(values) for key, values in metrics.items() if values}
        print(path)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
