"""Benchmark script for Python-only implementation.

Measures performance of the hot paths identified in the deep analysis:
1. Heap operations (add, pop, first, remove, heapsort)
2. EventQueue operations (add, step)
3. IntervalTree operations (add, remove, chop, merge_overlaps, iteration)
4. ResourcePool operations (find, allocate, free)
5. Cluster operations (find, find_resources_at_time, clone, state)
6. Full simulation run (end-to-end)
"""

import json
import statistics
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intervaltree import Interval, IntervalTree

from schedgym.cluster import Cluster
from schedgym.event import EventQueue, EventType, JobEvent
from schedgym.heap import Heap
from schedgym.job import Job
from schedgym.pool import ResourcePool, ResourceType
from schedgym.resource import Resource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bench(fn, *, warmup: int = 3, repeats: int = 20, label: str = ""):
    """Run *fn* repeatedly and return timing statistics (seconds)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    med = statistics.median(times)
    return {
        "label": label,
        "mean_s": mean,
        "median_s": med,
        "stdev_s": stdev,
        "min_s": min(times),
        "max_s": max(times),
        "repeats": repeats,
    }


# ---------------------------------------------------------------------------
# 1. Heap benchmarks
# ---------------------------------------------------------------------------


def bench_heap_add_pop(n: int = 10_000):
    """Add *n* items then pop them all."""

    def run():
        h = Heap()
        for i in range(n):
            h.add(i, priority=n - i)
        while len(h):
            h.pop()

    return bench(run, label=f"Heap add+pop ({n})")


def bench_heap_first_with_removals(n: int = 10_000):
    """Add *n* items, remove half (worst-case lazy-delete), then call first."""

    def run():
        h = Heap()
        for i in range(n):
            h.add(i, priority=i)
        # Remove every other item to create dead entries
        for i in range(0, n, 2):
            h.remove(i)
        # Repeated first calls force purging dead entries
        for _ in range(100):
            h.first

    return bench(run, label=f"Heap.first with 50% dead ({n})")


def bench_heap_heapsort(n: int = 10_000):
    """Iterate (heapsort) over a heap of *n* items."""

    def run():
        h = Heap()
        for i in range(n):
            h.add(i, priority=i)
        list(h)  # triggers heapsort via __iter__

    return bench(run, label=f"Heap heapsort ({n})")


# ---------------------------------------------------------------------------
# 2. EventQueue benchmarks
# ---------------------------------------------------------------------------


def _make_job(
    job_id: int, procs: int = 4, mem: int = 8, submit: int = 0, requested: int = 100
) -> Job:
    """Utility to create a Job with resource allocations."""
    j = Job()
    j.id = job_id
    j.submit_time = submit
    j.requested_time = requested
    j.requested_processors = procs
    j.requested_memory = mem
    j.resources = Resource(
        IntervalTree([Interval(0, procs, job_id)]),
        IntervalTree([Interval(0, mem, job_id)]),
    )
    return j


def bench_eventqueue_step(n_events: int = 5_000):
    """Add *n_events* future events then step through all of them."""

    def run():
        eq = EventQueue(time=0)
        for i in range(1, n_events + 1):
            job = _make_job(i, submit=0, requested=100)
            ev = JobEvent(time=i, type=EventType.JOB_START, job=job)
            eq.add(ev)
        # Step one unit at a time to exercise .first and pop
        for _ in range(n_events):
            eq.step(1)

    return bench(run, label=f"EventQueue add+step ({n_events})")


# ---------------------------------------------------------------------------
# 3. IntervalTree benchmarks
# ---------------------------------------------------------------------------


def bench_intervaltree_chop(n: int = 1_000):
    """Build a full interval, then chop *n* holes out of it."""

    def run():
        tree = IntervalTree([Interval(0, n * 10, None)])
        for i in range(n):
            tree.chop(i * 10 + 2, i * 10 + 8)

    return bench(run, label=f"IntervalTree chop ({n})")


def bench_intervaltree_add_merge(n: int = 5_000):
    """Add *n* overlapping intervals then merge."""

    def run():
        tree = IntervalTree()
        for i in range(n):
            tree.add(Interval(i, i + 5, i))
        tree.merge_overlaps()

    return bench(run, label=f"IntervalTree add+merge ({n})")


def bench_intervaltree_iteration(n: int = 5_000):
    """Iterate over a tree with *n* intervals."""

    def run():
        tree = IntervalTree([Interval(i * 2, i * 2 + 1, i) for i in range(n)])
        total = 0
        for iv in tree:
            total += iv.end - iv.begin

    return bench(run, label=f"IntervalTree iteration ({n})")


# ---------------------------------------------------------------------------
# 4. ResourcePool benchmarks
# ---------------------------------------------------------------------------


def bench_resourcepool_find_allocate_free(pool_size: int = 128, n_ops: int = 500):
    """Repeated find/allocate/free cycles on a pool."""

    def run():
        rp = ResourcePool(ResourceType.CPU, pool_size)
        for i in range(n_ops):
            size = (i % (pool_size // 4)) + 1
            intervals = rp.find(size, data=i)
            if intervals:
                rp.allocate(intervals)
                rp.free(intervals)

    return bench(
        run, label=f"ResourcePool find+alloc+free ({n_ops} ops, size={pool_size})"
    )


def bench_resourcepool_find_fragmented(pool_size: int = 256, n_allocs: int = 64):
    """Find in a heavily fragmented pool (worst case for chop)."""

    def run():
        rp = ResourcePool(ResourceType.CPU, pool_size)
        # Fragment the pool: allocate every other 2-unit slot
        allocated = []
        for i in range(0, pool_size, 4):
            iv = IntervalTree([Interval(i, i + 2, 999)])
            rp.allocate(iv)
            allocated.append(iv)
        # Now find a 3-unit slot (must navigate fragments)
        for _ in range(100):
            rp.find(3, data=0)
        # Clean up
        for iv in allocated:
            rp.free(iv)

    return bench(run, label=f"ResourcePool find fragmented ({pool_size})")


# ---------------------------------------------------------------------------
# 5. Cluster benchmarks
# ---------------------------------------------------------------------------


def bench_cluster_find(n_procs: int = 128, n_mem: int = 256, n_jobs: int = 200):
    """Find resources for jobs in a cluster."""

    def run():
        c = Cluster(n_procs, n_mem, ignore_memory=False)
        for i in range(n_jobs):
            procs = (i % 8) + 1
            mem = (i % 16) + 1
            j = _make_job(i, procs=procs, mem=mem)
            res = c.find(j)
            if res:
                j.resources = res
                c.allocate(j)
                c.free(j)

    return bench(run, label=f"Cluster find+alloc+free ({n_jobs} jobs)")


def bench_cluster_clone(n_procs: int = 64, n_mem: int = 128):
    """Clone a cluster with some allocations (tests deepcopy performance)."""

    def run():
        c = Cluster(n_procs, n_mem, ignore_memory=False)
        # Pre-allocate some resources to make clone non-trivial
        for i in range(10):
            j = _make_job(i, procs=4, mem=8)
            res = c.find(j)
            if res:
                j.resources = res
                c.allocate(j)
        # Clone 50 times (simulates inner-loop deepcopy)
        for _ in range(50):
            c.clone()

    return bench(run, label=f"Cluster clone (50x, {n_procs} procs)")


def bench_cluster_find_resources_at_time(n_procs: int = 64, n_mem: int = 128):
    """find_resources_at_time with a queue of events."""

    def run():
        c = Cluster(n_procs, n_mem, ignore_memory=False)
        # Create some allocated jobs
        jobs = []
        for i in range(8):
            j = _make_job(i, procs=4, mem=8, requested=100)
            res = c.find(j)
            if res:
                j.resources = res
                c.allocate(j)
                jobs.append(j)
        # Create future events
        events = []
        for i, j in enumerate(jobs):
            ev = JobEvent(time=i * 10 + 50, type=EventType.JOB_START, job=j)
            events.append(ev)
        # Find resources at various times
        target = _make_job(999, procs=8, mem=16, requested=50)
        for t in range(0, 200, 5):
            c.find_resources_at_time(t, target, events)

    return bench(run, label="Cluster find_resources_at_time (40 time steps)")


# ---------------------------------------------------------------------------
# 6. Full simulation benchmark
# ---------------------------------------------------------------------------


def bench_full_simulation(n_jobs: int = 500, n_procs: int = 64, n_mem: int = 128):
    """Simulate a simple job scheduling run: submit, find, allocate, step, free."""

    def run():
        cluster = Cluster(n_procs, n_mem, ignore_memory=True)
        eq = EventQueue(time=0)
        # Submit all jobs
        for i in range(n_jobs):
            j = _make_job(
                i, procs=(i % 8) + 1, mem=1, submit=i, requested=(i % 50) + 10
            )
            res = cluster.find(j)
            if res:
                j.resources = res
                cluster.allocate(j)
                finish_ev = JobEvent(
                    time=j.submit_time + j.requested_time,
                    type=EventType.JOB_FINISH,
                    job=j,
                )
                eq.add(finish_ev)
        # Step through time processing events
        current_time = 0
        while eq.next is not None:
            nxt = eq.next
            delta = nxt.time - current_time
            if delta <= 0:
                delta = 1
            happened = eq.step(delta)
            current_time = eq.time
            for ev in happened:
                if ev.type == EventType.JOB_FINISH:
                    cluster.free(ev.job)

    return bench(run, label=f"Full simulation ({n_jobs} jobs, {n_procs} procs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("  sched-rl-gym  Performance Baseline  (Python-only)")
    print("=" * 72)
    print()

    results = []

    benchmarks = [
        # Heap
        bench_heap_add_pop,
        bench_heap_first_with_removals,
        bench_heap_heapsort,
        # EventQueue
        bench_eventqueue_step,
        # IntervalTree
        bench_intervaltree_chop,
        bench_intervaltree_add_merge,
        bench_intervaltree_iteration,
        # ResourcePool
        bench_resourcepool_find_allocate_free,
        bench_resourcepool_find_fragmented,
        # Cluster
        bench_cluster_find,
        bench_cluster_clone,
        bench_cluster_find_resources_at_time,
        # Full simulation
        bench_full_simulation,
    ]

    for bm in benchmarks:
        print(f"  Running: {bm.__name__} ... ", end="", flush=True)
        r = bm()
        results.append(r)
        print(f"{r['mean_s'] * 1000:10.2f} ms  (±{r['stdev_s'] * 1000:.2f} ms)")

    print()
    print("-" * 72)
    print(f"  {'Benchmark':<55} {'Mean (ms)':>10}")
    print("-" * 72)
    for r in results:
        print(f"  {r['label']:<55} {r['mean_s'] * 1000:10.2f}")
    print("-" * 72)
    print()

    # Save results to JSON for later comparison
    out_path = Path(__file__).resolve().parent / "baseline_python.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
