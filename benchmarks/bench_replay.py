import argparse
import json
import time
from pathlib import Path

from schedgym import scheduler as schedulers
from schedgym.replay import ReplayConfig, TraceReplayEngine

SCHEDULERS = {
    "fifo": schedulers.FifoScheduler,
    "easy": schedulers.EasyScheduler,
    "backfilling": schedulers.BackfillingScheduler,
}


def run_once(args, limit: int):
    engine = TraceReplayEngine.from_swf(
        args.trace,
        ReplayConfig(
            scheduler_cls=SCHEDULERS[args.scheduler],
            processors=args.processors,
            memory=args.memory,
            ignore_memory=args.ignore_memory,
            trace_limit=limit,
            timeout_s=args.timeout,
        ),
    )
    result = engine.run()
    payload = {
        "trace": str(args.trace),
        "scheduler": args.scheduler,
        "limit": limit,
        "jobs_loaded": result.jobs_loaded,
        "jobs_completed": result.jobs_completed,
        "malformed_jobs_dropped": result.malformed_jobs_dropped,
        "simulated_end_time": result.simulated_end_time,
        "makespan": result.makespan,
        "avg_slowdown": result.avg_slowdown,
        "avg_bounded_slowdown": result.avg_bounded_slowdown,
        "wall_time_s": result.wall_time_s,
        "timeout_hit": result.timeout_hit,
        "event_count_processed": result.event_count_processed,
    }
    print(json.dumps(payload))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--scheduler", choices=sorted(SCHEDULERS), default="fifo")
    parser.add_argument("--processors", type=int, required=True)
    parser.add_argument("--memory", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--global-timeout", type=float, default=5400.0)
    parser.add_argument("--ignore-memory", action="store_true", default=False)
    parser.add_argument(
        "--ladder",
        type=int,
        nargs="*",
        default=None,
    )
    args = parser.parse_args()

    ladder = args.ladder or (
        [args.limit]
        if args.limit is not None
        else [1000, 5000, 10000, 25000, 50000, 100000]
    )
    campaign_start = time.perf_counter()
    for limit in ladder:
        if limit is None:
            continue
        for _ in range(args.repeats):
            if time.perf_counter() - campaign_start > args.global_timeout:
                print(json.dumps({"aborted": "global-timeout", "limit": limit}))
                return
            result = run_once(args, limit)
            if result.timeout_hit:
                print(json.dumps({"aborted": "run-timeout", "limit": limit}))
                return


if __name__ == "__main__":
    main()
