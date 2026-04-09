import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

from schedgym.replay import ReplayConfig, TraceReplayEngine
from schedgym.scheduler.fifo_scheduler import FifoScheduler


class TimedScheduler(FifoScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._csn_calls = 0
        self._csn_total_s = 0.0

    def can_schedule_now(self, job):
        t0 = time.perf_counter()
        result = super().can_schedule_now(job)
        self._csn_total_s += time.perf_counter() - t0
        self._csn_calls += 1
        return result


def run_profile(args) -> TimedScheduler:
    engine = TraceReplayEngine.from_swf(
        args.trace,
        ReplayConfig(
            scheduler_cls=TimedScheduler,
            processors=args.processors,
            memory=args.memory,
            ignore_memory=args.ignore_memory,
            trace_limit=args.jobs,
            timeout_s=args.timeout,
        ),
    )
    engine.run()
    return engine.scheduler


def print_summary(sched: TimedScheduler) -> None:
    calls = sched._csn_calls
    total_s = sched._csn_total_s
    avg_ms = (total_s / calls * 1000) if calls > 0 else 0.0

    print("=" * 60)
    print("  can_schedule_now() performance profile")
    print("=" * 60)
    print(f"  total_calls          : {calls:>12,}")
    print(f"  total_time_s         : {total_s:>12.4f} s")
    print(f"  avg_time_per_call_ms : {avg_ms:>12.4f} ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Profile can_schedule_now() performance via trace replay."
    )
    parser.add_argument(
        "--trace", type=Path, required=True, help="Path to SWF trace file"
    )
    parser.add_argument(
        "--jobs", type=int, default=5000, help="Number of jobs to replay"
    )
    parser.add_argument(
        "--processors", type=int, default=512, help="Cluster processor count"
    )
    parser.add_argument("--memory", type=int, default=512, help="Cluster memory size")
    parser.add_argument(
        "--ignore-memory",
        action="store_true",
        default=True,
        help="Ignore memory constraints (default: True)",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="Per-run timeout in seconds"
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        default=False,
        help="Run with cProfile and print top-10 hotspots",
    )
    args = parser.parse_args()

    if args.cprofile:
        pr = cProfile.Profile()
        pr.enable()
        sched = run_profile(args)
        pr.disable()
        print_summary(sched)
        print()
        print("Top-10 cProfile hotspots (cumulative time):")
        print("-" * 80)
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(10)
        print(s.getvalue())
    else:
        sched = run_profile(args)
        print_summary(sched)


if __name__ == "__main__":
    main()
