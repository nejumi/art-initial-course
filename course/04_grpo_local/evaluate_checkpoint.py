from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio
from pathlib import Path

from course.shared.art_compat import make_local_backend, make_trainable_model
from course.shared.config import config_from_env
from course.shared.data import load_cached_split, scenarios_from_records, write_jsonl, write_sample_dataset
from course.shared.rollout import rollout_retail
from course.shared.tracing import init_weave


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the current ART checkpoint on retail scenarios.")
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="data/retail/eval_current.jsonl")
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument("--weave", action="store_true", help="Trace eval rollouts to Weave during this run.")
    args = parser.parse_args()

    cfg = config_from_env()
    if args.weave:
        init_weave(cfg.project)
    data_dir = Path(args.data_dir)
    if not (data_dir / f"{args.split}.jsonl").exists():
        write_sample_dataset(data_dir)
    records = load_cached_split(data_dir, args.split, limit=args.limit)
    scenarios = scenarios_from_records(records, split=args.split)

    backend = make_local_backend(cfg.art_path)
    model = make_trainable_model(cfg)
    await model.register(backend)

    import art

    groups = []
    rows = []
    for scenario in scenarios:
        traj = await rollout_retail(model, scenario, split="val", temperature=0.2, request_logprobs=not args.no_logprobs)
        groups.append(art.TrajectoryGroup([traj]))
        rows.append({"scenario_id": scenario.id, "reward": traj.reward, "metrics": traj.metrics, "logs": traj.logs})
        print(scenario.id, traj.reward, traj.metrics)
    await model.log(groups, split="val")
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} eval rows to {args.output}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
